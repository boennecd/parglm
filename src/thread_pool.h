/*
 Boost Software License - Version 1.0 - August 17th, 2003

 Permission is hereby granted, free of charge, to any person or organization
 obtaining a copy of the software and accompanying documentation covered by
 this license (the "Software") to use, reproduce, display, distribute,
 execute, and transmit the Software, and to prepare derivative works of the
 Software, and to permit third-parties to whom the Software is furnished to
 do so, all subject to the following:

 The copyright notices in the Software and this entire statement, including
 the above license grant, this restriction and the following disclaimer,
 must be included in all copies of the Software, in whole or in part, and
 all derivative works of the Software, unless such copies or derivative
 works are solely in the form of machine-executable object code generated by
 a source language processor.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
 */

/*
 Code from:
 Williams, Anthony. C++ concurrency in action. London, 2012
 Available at: https://www.manning.com/books/c-plus-plus-concurrency-in-action
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <deque>
#include <future>
#include <memory>
#include <functional>
#include <iostream>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <type_traits>

/*
 Listing 6.6 A thread-safe queue with fine-grained locking
 Uses a singly linked list with a seperate mutex for the head and tail
 */
template<typename T>
class thread_safe_queue
{
private:
   struct node
   {
      std::shared_ptr<T> data;
      std::unique_ptr<node> next;
   };

   std::mutex head_mutex;
   std::unique_ptr<node> head;
   std::mutex tail_mutex;
   node* tail;

   node* get_tail()
   {
      std::lock_guard<std::mutex> tail_lock(tail_mutex);
      return tail;
   }

   std::unique_ptr<node> pop_head()
   {
      std::lock_guard<std::mutex> head_lock(head_mutex);
      if (head.get() == get_tail())
      {
         return nullptr;
      }
      std::unique_ptr<node> old_head = std::move(head);
      head = std::move(old_head->next);
      return old_head;
   }


public:
   thread_safe_queue() :
      head(new node), tail(head.get())
   {}

   thread_safe_queue(const thread_safe_queue& other) = delete;
   thread_safe_queue& operator=(const thread_safe_queue& other) = delete;

   std::shared_ptr<T> try_pop()
   {
      std::unique_ptr<node> old_head = pop_head();
      return old_head ? old_head->data : std::shared_ptr<T>();
   }

   // Added
   bool try_pop(T& value)
   {
      std::unique_ptr<node> old_head = pop_head();
      if(old_head){
        value = std::move(*old_head->data);
        return true;

      } else{
        return false;

      }
   }

   void push(T new_value)
   {
      std::shared_ptr<T> new_data(
         std::make_shared<T>(std::move(new_value)));
      std::unique_ptr<node> p(new node);
      node* const new_tail = p.get();
      std::lock_guard<std::mutex> tail_lock(tail_mutex);
      tail->data = new_data;
      tail->next = std::move(p);
      tail = new_tail;
   }
};

// Just before listing 8.4
class join_threads
{
  std::vector<std::thread>& threads;
public:
  explicit join_threads(std::vector<std::thread>& threads_);
  ~join_threads();
};

// Listing 9.2
class function_wrapper
{
  struct impl_base {
    virtual void call()=0;
    virtual ~impl_base() {}
  };
  std::unique_ptr<impl_base> impl;
  template<typename F>
  struct impl_type: impl_base
  {
    F f;
    impl_type(F&& f_): f(std::move(f_)) {}
    void call() { f(); }
  };
public:
  template<typename F>
  function_wrapper(F&& f):
    impl(new impl_type<F>(std::move(f)))
  {}

  void operator()() { impl->call(); }
  function_wrapper() = default;
  function_wrapper(function_wrapper&& other):
    impl(std::move(other.impl))
  {}

  function_wrapper& operator=(function_wrapper&& other)
  {
    impl=std::move(other.impl);
    return *this;
  }

  bool has_value () const {
    return (bool)impl;
  }

  function_wrapper(const function_wrapper&)=delete;
  function_wrapper(function_wrapper&)=delete;
  function_wrapper& operator=(const function_wrapper&)=delete;
};

// Listing 9.2:
class thread_pool
{
  thread_safe_queue<function_wrapper> work_queue;
  std::condition_variable cv;
  std::mutex mu;

  void worker_thread()
  {
    for(;;){
      function_wrapper task;

      bool got_task = work_queue.try_pop(task);
      if(!got_task){
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]{ return work_queue.try_pop(task) or done; });

        if(done and !task.has_value())
          return;
      }

      task();
    }
  }

  // From listing 9.1
  std::atomic_bool done;
  std::vector<std::thread> threads;
  join_threads joiner;

  // Added
  unsigned const thread_count;

public:
  template<typename FunctionType>
  std::future<typename std::result_of<FunctionType()>::type>
  submit(FunctionType f)
  {
    typedef typename std::result_of<FunctionType()>::type result_type;

    std::packaged_task<result_type()> task(std::move(f));
    std::future<result_type> res(task.get_future());
    work_queue.push(std::move(task));
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.notify_one();
    }
    return res;
  }

  // From listing 9.2
  thread_pool(unsigned const n_threads = 1):
    done(false),
    joiner(threads),
    thread_count(n_threads)
  {
    // Moved to private member
    //unsigned const thread_count=std::thread::hardware_concurrency();
    try
    {
      for(unsigned i=0;i<thread_count;++i)
      {
        threads.push_back(
          std::thread(&thread_pool::worker_thread,this));
      }
    }
    catch(...)
    {
      {
        std::unique_lock<std::mutex> lk(mu);
        done=true;
      }
      cv.notify_all();
      throw;
    }
  }

  ~thread_pool()
  {
    {
      std::unique_lock<std::mutex> lk(mu);
      done=true;
    }
    cv.notify_all();
  }
};

#endif

#ifndef DDFAMILY
#define DDFAMILY
#include "arma_n_rcpp.h"
#include <memory>

class glm_base {
public:
  virtual double dev_resids(double, double, double) const = 0;
  virtual double dev_resids
    (const arma::vec &, const arma::vec &, const arma::vec&) const = 0;

  virtual double linkfun(double) const = 0;
  virtual void linkfun(arma::vec&, const arma::vec&) const = 0;
  arma::vec linkfun(const arma::vec &inp) const {
    arma::vec out(inp.n_elem);
    linkfun(out, inp);
    return out;
  }

  virtual double linkinv(double) const = 0;
  virtual void linkinv(arma::vec&, const arma::vec&) const = 0;
  arma::vec linkinv(const arma::vec &inp) const {
    arma::vec out(inp.n_elem);
    linkinv(out, inp);
    return out;
  }

  virtual double variance(double) const = 0;
  virtual void variance(arma::vec&, const arma::vec&) const = 0;
  arma::vec variance(const arma::vec &inp) const {
    arma::vec out(inp.n_elem);
    variance(out, inp);
    return out;
  }

  virtual double mu_eta(double) const = 0;
  virtual void mu_eta(arma::vec&, const arma::vec&) const = 0;
  arma::vec mu_eta(const arma::vec &inp) const {
    arma::vec out(inp.n_elem);
    mu_eta(out, inp);
    return out;
  }

  virtual double initialize(double, double) const = 0;
  virtual void initialize
    (arma::vec&, const arma::vec&, const arma::vec&) const = 0;

  virtual std::string name() const = 0;

  // create a virtual, default destructor
  virtual ~glm_base() = default;
};

std::unique_ptr<glm_base> get_fam_obj(const std::string family);

#define GLM_CLASS(fname)                                       \
  class fname final : public glm_base {                        \
    public:                                                    \
      double dev_resids(double, double, double) const;         \
      double dev_resids                                        \
        (const arma::vec &y, const arma::vec &mu,              \
         const arma::vec &wt) const                            \
      {                                                        \
        const double *m = mu.begin(), *w = wt.begin();         \
        double out = 0.;                                       \
        for(auto yi = y.begin(); yi != y.end(); ++yi, ++m, ++w)\
          out += dev_resids(*yi, *m, *w);                      \
                                                               \
        return out;                                            \
      }                                                        \
      double linkfun(double) const;                            \
      void linkfun(arma::vec &out, const arma::vec& arg) const \
      {                                                        \
        const double *a{arg.begin()};                          \
        for(auto o = out.begin(); o != out.end();)             \
          *o++ = linkfun(*a++);                                \
      }                                                        \
      double linkinv(double) const;                            \
      void linkinv(arma::vec &out, const arma::vec& arg) const \
      {                                                        \
        const double *a{arg.begin()};                          \
        for(auto o = out.begin(); o != out.end();)             \
          *o++ = linkinv(*a++);                                \
      }                                                        \
      double variance(double) const;                           \
      void variance(arma::vec &out, const arma::vec& arg) const\
      {                                                        \
        const double *a{arg.begin()};                          \
        for(auto o = out.begin(); o != out.end();)             \
          *o++ = variance(*a++);                               \
      }                                                        \
      double mu_eta(double) const;                             \
      void mu_eta(arma::vec &out, const arma::vec& arg) const  \
      {                                                        \
        const double *a{arg.begin()};                          \
        for(auto o = out.begin(); o != out.end();)             \
          *o++ = mu_eta(*a++);                                 \
      }                                                        \
      double initialize(double, double) const;                 \
      void initialize                                          \
        (arma::vec &eta, const arma::vec &y,                   \
         const arma::vec &wt) const {                          \
        const double *yi = y.begin(), *w = wt.begin();         \
        for(auto e = eta.begin(); e != eta.end();              \
            ++e, ++yi, ++w)                                    \
          *e = initialize(*yi, *w);                            \
      }                                                        \
      std::string name() const;                                \
    };

GLM_CLASS(binomial_logit)
GLM_CLASS(binomial_probit)
GLM_CLASS(binomial_cauchit)
GLM_CLASS(binomial_log)
GLM_CLASS(binomial_cloglog)

GLM_CLASS(gaussian_identity)
GLM_CLASS(gaussian_log)
GLM_CLASS(gaussian_inverse)

GLM_CLASS(Gamma_inverse)
GLM_CLASS(Gamma_identity)
GLM_CLASS(Gamma_log)

GLM_CLASS(poisson_log)
GLM_CLASS(poisson_identity)
GLM_CLASS(poisson_sqrt)

GLM_CLASS(inverse_gaussian_1_mu_2)
GLM_CLASS(inverse_gaussian_inverse)
GLM_CLASS(inverse_gaussian_identity)
GLM_CLASS(inverse_gaussian_log)

#undef GLM_CLASS
#endif

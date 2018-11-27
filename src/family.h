#ifndef DDFAMILY
#define DDFAMILY
#include "arma_n_rcpp.h"
#include <memory>

#define VEC_OVERLOAD(fname)                                           \
  inline void fname(arma::vec &out, const arma::vec &inp) const {     \
    if(out.n_elem != inp.n_elem)                                      \
      Rcpp::stop("Unequal length of vectors");                        \
                                                                      \
    const double *x = inp.begin();                                    \
    for(auto o = out.begin(); o != out.end(); ++o, ++x)               \
      *o = fname(*x);                                                 \
  }                                                                   \
  inline arma::vec fname(const arma::vec &inp) const {                \
    arma::vec out(inp.n_elem);                                        \
    fname(out, inp);                                                  \
    return out;                                                       \
  }

class glm_base {
public:
  virtual double dev_resids(double, double, double) const = 0;

  double dev_resids(
      const arma::vec &y, const arma::vec &mu, const arma::vec &wt) const
  {
    if(y.n_elem != mu.n_elem or y.n_elem != wt.n_elem)
      Rcpp::stop("Unequal length of vectors");

    const double *m = mu.begin(), *w = wt.begin();
    double out = 0.;
    for(auto yi = y.begin(); yi != y.end(); ++yi, ++m, ++w)
      out += dev_resids(*yi, *m, *w);

    return out;
  }

  virtual double linkfun(double) const = 0;
  VEC_OVERLOAD(linkfun)

  virtual double linkinv(double) const = 0;
  VEC_OVERLOAD(linkinv)

  virtual double variance(double) const = 0;
  VEC_OVERLOAD(variance)

  virtual double mu_eta(double) const = 0;
  VEC_OVERLOAD(mu_eta)

  virtual double initialize(double, double) const = 0;

  void initialize(
      arma::vec &eta, const arma::vec &y, const arma::vec &wt) const
  {
    if(eta.n_elem != y.n_elem or eta.n_elem != wt.n_elem)
      Rcpp::stop("Unequal length of vectors");

    const double *yi = y.begin(), *w = wt.begin();
    for(auto e = eta.begin(); e != eta.end(); ++e, ++yi, ++w)
      *e = initialize(*yi, *w);
  }

  virtual std::string name() const = 0;

  // create a virtual, default destructor
  virtual ~glm_base() = default;
};

std::unique_ptr<glm_base> get_fam_obj(const std::string family);

#define GLM_CLASS(fname)                                            \
  class fname : public glm_base {                                   \
    public:                                                         \
      double dev_resids(double, double, double) const override; \
      double linkfun(double) const override;                    \
      double linkinv(double) const override;                    \
      double variance(double) const override;                   \
      double mu_eta(double) const override;                     \
      double initialize(double, double) const override;         \
      std::string name() const override;                        \
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
#undef VEC_OVERLOAD
#endif

#include "arma_n_rcpp.h"
#include "family.h"
#include <float.h>
#include <limits>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static const double THRESH = 30.;
static const double MTHRESH = -30.;

inline double binomial_dev_resids(double y, double mu, double wt){
  return - 2 * wt * (y * log(mu) + (1 - y) * log(1 - mu));
}

/*----------------------------------------------------------------------------*/

double binomial_logit::linkfun(double mu) const {
  return log(mu / (1 - mu));
}

double binomial_logit::linkinv(double eta) const {
  double tmp = (eta < MTHRESH) ? DBL_EPSILON :
    ((eta > THRESH) ? 1 / DBL_EPSILON  : exp(eta));
  return tmp / (1 + tmp);
}

double binomial_logit::variance(double mu) const {
  return mu * (1 - mu);
}

double binomial_logit::dev_resids(double y, double mu, double wt) const {
  return binomial_dev_resids(y, mu, wt);
}

double binomial_logit::mu_eta(double eta) const {
  double opexp = 1 + exp(eta);

  return (eta > THRESH || eta < MTHRESH) ? DOUBLE_EPS :
    exp(eta)/(opexp * opexp);
}

double binomial_logit::initialize(double y, double weight) const {
  return linkfun((weight * y + 0.5)/(weight + 1));
}


std::string binomial_logit::name() const {
  return "binomial_logit";
}

/*----------------------------------------------------------------------------*/

double binomial_probit::linkfun(double mu) const {
  return R::qnorm(mu, 0, 1, 1, 0);
}

double binomial_probit::linkinv(double eta) const {
  const double thresh =
    -R::qnorm(std::numeric_limits<double>::epsilon(), 0, 1, 1, 0);
  eta = MIN(MAX(eta, -thresh), thresh);
  return R::pnorm(eta, 0, 1, 1, 0);
}

double binomial_probit::variance(double mu) const {
  return mu * (1 - mu);
}

double binomial_probit::dev_resids(double y, double mu, double wt) const {
  return binomial_dev_resids(y, mu, wt);
}

double binomial_probit::mu_eta(double eta) const {
  return MAX(R::dnorm(eta, 0, 1, 0), std::numeric_limits<double>::epsilon());
}

double binomial_probit::initialize(double y, double weight) const {
  return linkfun((weight * y + 0.5)/(weight + 1));
}

std::string binomial_probit::name() const {
  return "binomial_probit";
}

/*----------------------------------------------------------------------------*/

double binomial_cauchit::linkfun(double mu) const {
  return R::qcauchy(mu, 0, 1, 1, 0);
}

double binomial_cauchit::linkinv(double eta) const {
  const double thresh =
    -R::qcauchy(std::numeric_limits<double>::epsilon(), 0, 1, 1, 0);
  eta = MIN(MAX(eta, -thresh), thresh);
  return R::pcauchy(eta, 0, 1, 1, 0);
}

double binomial_cauchit::variance(double mu) const {
  return mu * (1 - mu);
}

double binomial_cauchit::dev_resids(double y, double mu, double wt) const {
  return binomial_dev_resids(y, mu, wt);
}

double binomial_cauchit::mu_eta(double eta) const {
  return MAX(R::dcauchy(eta, 0, 1, 0), std::numeric_limits<double>::epsilon());
}

double binomial_cauchit::initialize(double y, double weight) const {
  return linkfun((weight * y + 0.5)/(weight + 1));
}

std::string binomial_cauchit::name() const {
  return "binomial_cauchit";
}

/*----------------------------------------------------------------------------*/

double binomial_log::linkfun(double mu) const {
  return std::log(mu);
}

double binomial_log::linkinv(double eta) const {
  return MAX(std::exp(eta), std::numeric_limits<double>::epsilon());
}

double binomial_log::variance(double mu) const {
  return mu * (1 - mu);
}

double binomial_log::dev_resids(double y, double mu, double wt) const {
  return binomial_dev_resids(y, mu, wt);
}

double binomial_log::mu_eta(double eta) const {
  return linkinv(eta);
}

double binomial_log::initialize(double y, double weight) const {
  return linkfun((weight * y + 0.5)/(weight + 1));
}

std::string binomial_log::name() const {
  return "binomial_log";
}

/*----------------------------------------------------------------------------*/

double binomial_cloglog::linkfun(double mu) const {
  return std::log(-std::log1p(-mu));
}

double binomial_cloglog::linkinv(double eta) const {
  return MAX(MIN(-std::expm1(-std::exp(eta)),
                 1 - std::numeric_limits<double>::epsilon()),
             std::numeric_limits<double>::epsilon());
}

double binomial_cloglog::variance(double mu) const {
  return mu * (1 - mu);
}

double binomial_cloglog::dev_resids(double y, double mu, double wt) const {
  return binomial_dev_resids(y, mu, wt);
}

double binomial_cloglog::mu_eta(double eta) const {
  eta = MIN(eta, 700);
  double f = exp(eta);
  return MAX(f * exp(-f), std::numeric_limits<double>::epsilon());
}

double binomial_cloglog::initialize(double y, double weight) const {
  return linkfun((weight * y + 0.5)/(weight + 1));
}

std::string binomial_cloglog::name() const {
  return "binomial_cloglog";
}

/*----------------------------------------------------------------------------*/





std::unique_ptr<glm_base> get_fam_obj(const std::string family){
  if(family == "binomial_logit")
    return std::unique_ptr<glm_base>(new binomial_logit());
  if(family == "binomial_probit")
    return std::unique_ptr<glm_base>(new binomial_probit());
  if(family == "binomial_cauchit")
    return std::unique_ptr<glm_base>(new binomial_cauchit());
  if(family == "binomial_log")
    return std::unique_ptr<glm_base>(new binomial_log());
  if(family == "binomial_cloglog")
    return std::unique_ptr<glm_base>(new binomial_cloglog());

  Rcpp::stop("family and link '" + family + "' is not supported");
}

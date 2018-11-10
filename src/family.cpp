#include "arma_n_rcpp.h"
#include "family.h"
#include <float.h>

static const double THRESH = 30.;
static const double MTHRESH = -30.;

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
  return - 2 * wt * (y * log(mu) + (1 - y) * log(1 - mu));
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




std::unique_ptr<glm_base> get_fam_obj(const std::string family){
  if(family == "binomial_logit")
    return std::unique_ptr<glm_base>(new binomial_logit());

  Rcpp::stop("family and link '" + family + "' is not supported");
}

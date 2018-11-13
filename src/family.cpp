#include "arma_n_rcpp.h"
#include "family.h"
#include <float.h>
#include <limits>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static const double THRESH = 30.;
static const double MTHRESH = -30.;

inline double log_linkinv(double eta){
  return MAX(std::exp(eta), std::numeric_limits<double>::epsilon());
}

inline double log_mu_eta(double eta){
  return MAX(std::exp(eta), std::numeric_limits<double>::epsilon());
}


/*----------------------------------------------------------------------------*/

inline double binomial_dev_resids(double y, double mu, double wt){
  return - 2 * wt * (y * log(mu) + (1 - y) * log(1 - mu));
}

inline double binomial_var(double mu){
  return mu * (1 - mu);
}

double binomial_logit::linkfun(double mu) const {
  return std::log(mu / (1 - mu));
}

double binomial_logit::linkinv(double eta) const {
  double tmp = (eta < MTHRESH) ? DBL_EPSILON :
    ((eta > THRESH) ? 1 / DBL_EPSILON  : exp(eta));
  return tmp / (1 + tmp);
}

double binomial_logit::variance(double mu) const {
  return binomial_var(mu);
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
  return binomial_var(mu);
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
  return binomial_var(mu);
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
  return log_linkinv(eta);
}

double binomial_log::variance(double mu) const {
  return binomial_var(mu);;
}

double binomial_log::dev_resids(double y, double mu, double wt) const {
  return binomial_dev_resids(y, mu, wt);
}

double binomial_log::mu_eta(double eta) const {
  return log_mu_eta(eta);
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
  return binomial_var(mu);
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

inline double gaussian_dev_resids(double y, double mu, double wt){
  double diff = y - mu;
  return wt * (diff * diff);
}

double gaussian_identity::linkfun(double mu) const {
  return mu;
}

double gaussian_identity::linkinv(double eta) const {
  return eta;
}

double gaussian_identity::variance(double mu) const {
  return 1.;
}

double gaussian_identity::dev_resids(double y, double mu, double wt) const {
  return gaussian_dev_resids(y, mu, wt);
}

double gaussian_identity::mu_eta(double eta) const {
  return 1;
}

double gaussian_identity::initialize(double y, double weight) const {
  return y;
}


std::string gaussian_identity::name() const {
  return "gaussian_identity";
}

/*----------------------------------------------------------------------------*/

double gaussian_log::linkfun(double mu) const {
  return std::log(mu);
}

double gaussian_log::linkinv(double eta) const {
  return log_linkinv(eta);
}

double gaussian_log::variance(double mu) const {
  return 1.;
}

double gaussian_log::dev_resids(double y, double mu, double wt) const {
  return gaussian_dev_resids(y, mu, wt);
}

double gaussian_log::mu_eta(double eta) const {
  return log_mu_eta(eta);
}

double gaussian_log::initialize(double y, double weight) const {
  if(y <= 0)
    Rcpp::stop("cannot find valid starting values: please specify some");
  return linkfun(y);
}

std::string gaussian_log::name() const {
  return "gaussian_log";
}

/*----------------------------------------------------------------------------*/

double gaussian_inverse::linkfun(double mu) const {
  return 1./mu;
}

double gaussian_inverse::linkinv(double eta) const {
  return 1./eta;
}

double gaussian_inverse::variance(double mu) const {
  return 1.;
}

double gaussian_inverse::dev_resids(double y, double mu, double wt) const {
  return gaussian_dev_resids(y, mu, wt);
}

double gaussian_inverse::mu_eta(double eta) const {
  return - 1. / (eta * eta);
}

double gaussian_inverse::initialize(double y, double weight) const {
  if(y == 0)
    Rcpp::stop("cannot find valid starting values: please specify some");
  return linkfun(y);
}


std::string gaussian_inverse::name() const {
  return "gaussian_inverse";
}

/*----------------------------------------------------------------------------*/

inline double poisson_dev_resids(double y, double mu, double wt){
  double res = (y > 0) ? y * std::log(y / mu) - (y - mu) : mu * wt;
  return 2 * res;
}

double poisson_log::linkfun(double mu) const {
  return std::log(mu);
}

double poisson_log::linkinv(double eta) const {
  return log_linkinv(eta);
}

double poisson_log::variance(double mu) const {
  return mu;
}

double poisson_log::dev_resids(double y, double mu, double wt) const {
  return poisson_dev_resids(y, mu, wt);
}

double poisson_log::mu_eta(double eta) const {
  return log_mu_eta(eta);
}

double poisson_log::initialize(double y, double weight) const {
  return linkfun(y + .1);
}


std::string poisson_log::name() const {
  return "poisson_log";
}

/*----------------------------------------------------------------------------*/

double poisson_identity::linkfun(double mu) const {
  return mu;
}

double poisson_identity::linkinv(double eta) const {
  return eta;
}

double poisson_identity::variance(double mu) const {
  return mu;
}

double poisson_identity::dev_resids(double y, double mu, double wt) const {
  return poisson_dev_resids(y, mu, wt);
}

double poisson_identity::mu_eta(double eta) const {
  return 1.;
}

double poisson_identity::initialize(double y, double weight) const {
  return y + .1;
}


std::string poisson_identity::name() const {
  return "poisson_identity";
}

/*----------------------------------------------------------------------------*/

double poisson_sqrt::linkfun(double mu) const {
  return std::sqrt(mu);
}

double poisson_sqrt::linkinv(double eta) const {
  return eta * eta;
}

double poisson_sqrt::variance(double mu) const {
  return mu;
}

double poisson_sqrt::dev_resids(double y, double mu, double wt) const {
  return poisson_dev_resids(y, mu, wt);
}

double poisson_sqrt::mu_eta(double eta) const {
  return 2. * eta;
}

double poisson_sqrt::initialize(double y, double weight) const {
  return linkfun(y + .1);
}


std::string poisson_sqrt::name() const {
  return "poisson_sqrt";
}

/*----------------------------------------------------------------------------*/

inline double inverse_gaussian_resids(double y, double mu, double wt){
  double d1 = y - mu;
  return wt * ((d1 * d1) / (y * mu * mu));
}

double inverse_gaussian_1_mu_2::linkfun(double mu) const {
  return 1 / (mu * mu);
}

double inverse_gaussian_1_mu_2::linkinv(double eta) const {
  return 1 / std::sqrt(eta);
}

double inverse_gaussian_1_mu_2::variance(double mu) const {
  return mu * mu * mu;
}

double inverse_gaussian_1_mu_2::dev_resids(double y, double mu, double wt) const {
  return inverse_gaussian_resids(y, mu, wt);
}

double inverse_gaussian_1_mu_2::mu_eta(double eta) const {
  return -1/(2 * std::pow(eta, 1.5));
}

double inverse_gaussian_1_mu_2::initialize(double y, double weight) const {
  if(y <= 0.)
    Rcpp::stop("positive values only are allowed for the 'inverse.gaussian' family");
  return linkfun(y);
}


std::string inverse_gaussian_1_mu_2::name() const {
  return "inverse_gaussian_1_mu_2";
}

/*----------------------------------------------------------------------------*/

double inverse_gaussian_inverse::linkfun(double mu) const {
  return 1 / mu;
}

double inverse_gaussian_inverse::linkinv(double eta) const {
  return 1 / eta;
}

double inverse_gaussian_inverse::variance(double mu) const {
  return mu * mu * mu;
}

double inverse_gaussian_inverse::dev_resids(double y, double mu, double wt) const {
  return inverse_gaussian_resids(y, mu, wt);
}

double inverse_gaussian_inverse::mu_eta(double eta) const {
  return -1/(eta * eta);
}

double inverse_gaussian_inverse::initialize(double y, double weight) const {
  if(y <= 0.)
    Rcpp::stop("positive values only are allowed for the 'inverse.gaussian' family");
  return linkfun(y);
}


std::string inverse_gaussian_inverse::name() const {
  return "inverse_gaussian_inverse";
}

/*----------------------------------------------------------------------------*/

double inverse_gaussian_identity::linkfun(double mu) const {
  return mu;
}

double inverse_gaussian_identity::linkinv(double eta) const {
  return eta;
}

double inverse_gaussian_identity::variance(double mu) const {
  return mu * mu * mu;
}

double inverse_gaussian_identity::dev_resids(double y, double mu, double wt) const {
  return inverse_gaussian_resids(y, mu, wt);
}

double inverse_gaussian_identity::mu_eta(double eta) const {
  return 1.;
}

double inverse_gaussian_identity::initialize(double y, double weight) const {
  if(y <= 0.)
    Rcpp::stop("positive values only are allowed for the 'inverse.gaussian' family");
  return linkfun(y);
}

std::string inverse_gaussian_identity::name() const {
  return "inverse_gaussian_inverse";
}

/*----------------------------------------------------------------------------*/

double inverse_gaussian_log::linkfun(double mu) const {
  return std::log(mu);
}

double inverse_gaussian_log::linkinv(double eta) const {
  return log_linkinv(eta);
}

double inverse_gaussian_log::variance(double mu) const {
  return mu * mu * mu;
}

double inverse_gaussian_log::dev_resids(double y, double mu, double wt) const {
  return inverse_gaussian_resids(y, mu, wt);
}

double inverse_gaussian_log::mu_eta(double eta) const {
  return log_mu_eta(eta);
}

double inverse_gaussian_log::initialize(double y, double weight) const {
  if(y <= 0.)
    Rcpp::stop("positive values only are allowed for the 'inverse.gaussian' family");
  return linkfun(y);
}

std::string inverse_gaussian_log::name() const {
  return "inverse_gaussian_log";
}

/*----------------------------------------------------------------------------*/

inline double Gamma_dev_resids(double y, double mu, double wt){
  double f = (y > 0) ? y / mu : 1;
  return -2 * wt * (std::log(f) - (y - mu)/mu);
}

double Gamma_inverse::linkfun(double mu) const {
  return 1/mu;
}

double Gamma_inverse::linkinv(double eta) const {
  return 1 / eta;
}

double Gamma_inverse::variance(double mu) const {
  return mu * mu;
}

double Gamma_inverse::dev_resids(double y, double mu, double wt) const {
  return Gamma_dev_resids(y, mu, wt);
}

double Gamma_inverse::mu_eta(double eta) const {
  return -1 / (eta * eta);
}

double Gamma_inverse::initialize(double y, double weight) const {
  if(y <= 0)
    Rcpp::stop("non-positive values not allowed for the 'gamma' family");
  return linkfun(y);
}


std::string Gamma_inverse::name() const {
  return "Gamma_inverse";
}

/*----------------------------------------------------------------------------*/

double Gamma_identity::linkfun(double mu) const {
  return mu;
}

double Gamma_identity::linkinv(double eta) const {
  return eta;
}

double Gamma_identity::variance(double mu) const {
  return mu * mu;
}

double Gamma_identity::dev_resids(double y, double mu, double wt) const {
  return Gamma_dev_resids(y, mu, wt);
}

double Gamma_identity::mu_eta(double eta) const {
  return .1;
}

double Gamma_identity::initialize(double y, double weight) const {
  if(y <= 0)
    Rcpp::stop("non-positive values not allowed for the 'gamma' family");
  return linkfun(y);
}


std::string Gamma_identity::name() const {
  return "Gamma_identity";
}

/*----------------------------------------------------------------------------*/

double Gamma_log::linkfun(double mu) const {
  return std::log(mu);
}

double Gamma_log::linkinv(double eta) const {
  return log_linkinv(eta);
}

double Gamma_log::variance(double mu) const {
  return mu * mu;
}

double Gamma_log::dev_resids(double y, double mu, double wt) const {
  return Gamma_dev_resids(y, mu, wt);
}

double Gamma_log::mu_eta(double eta) const {
  return log_mu_eta(eta);
}

double Gamma_log::initialize(double y, double weight) const {
  if(y <= 0)
    Rcpp::stop("non-positive values not allowed for the 'gamma' family");
  return linkfun(y);
}


std::string Gamma_log::name() const {
  return "Gamma_log";
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

  if(family == "gaussian_identity")
    return std::unique_ptr<glm_base>(new gaussian_identity());
  if(family == "gaussian_log")
    return std::unique_ptr<glm_base>(new gaussian_log());
  if(family == "gaussian_inverse")
    return std::unique_ptr<glm_base>(new gaussian_inverse());

  if(family == "Gamma_inverse")
    return std::unique_ptr<glm_base>(new Gamma_inverse());
  if(family == "Gamma_identity")
    return std::unique_ptr<glm_base>(new Gamma_identity());
  if(family == "Gamma_log")
    return std::unique_ptr<glm_base>(new Gamma_log());

  if(family == "poisson_log")
    return std::unique_ptr<glm_base>(new poisson_log());
  if(family == "poisson_identity")
    return std::unique_ptr<glm_base>(new poisson_identity());
  if(family == "poisson_sqrt")
    return std::unique_ptr<glm_base>(new poisson_sqrt());

  if(family == "inverse.gaussian_1_mu_2")
    return std::unique_ptr<glm_base>(new inverse_gaussian_1_mu_2());
  if(family == "inverse.gaussian_inverse")
    return std::unique_ptr<glm_base>(new inverse_gaussian_inverse());
  if(family == "inverse.gaussian_identity")
    return std::unique_ptr<glm_base>(new inverse_gaussian_identity());
  if(family == "inverse.gaussian_log")
    return std::unique_ptr<glm_base>(new inverse_gaussian_log());

  Rcpp::stop("family and link '" + family + "' is not supported");
}

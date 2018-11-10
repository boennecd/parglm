#ifndef DDFAMILY
#define DDFAMILY

#include <memory>

class glm_base {
public:
  virtual double dev_resids(double, double, double) const = 0;

  virtual double linkfun(double) const = 0;

  virtual double linkinv(double) const = 0;

  virtual double variance(double) const = 0;

  virtual double mu_eta(double) const = 0;

  virtual double initialize(double, double) const = 0;

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
      std::string name() const override;                            \
    };

GLM_CLASS(binomial_logit)
GLM_CLASS(binomial_probit)
GLM_CLASS(binomial_cauchit)
GLM_CLASS(binomial_log)
GLM_CLASS(binomial_cloglog)

#endif

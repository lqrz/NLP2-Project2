#pragma once

#include <string>
#include "StatelessFeatureFunction.h"

namespace Moses
{
class TargetPhrase;
class ScoreComponentCollection;

class Word2VecFeature : public StatelessFeatureFunction
{
protected:
  static Word2VecFeature *s_instance;

public:
  static const Word2VecFeature& Instance() {
    return *s_instance;
  }
  static Word2VecFeature& InstanceNonConst() {
    return *s_instance;
  }

  Word2VecFeature(const std::string &line);

  bool IsUseable(const FactorMask &mask) const {
    return true;
  }

  virtual void EvaluateInIsolation(const Phrase &source
                                   , const TargetPhrase &targetPhrase
                                   , ScoreComponentCollection &scoreBreakdown
                                   , ScoreComponentCollection &estimatedFutureScore) const;
  void EvaluateWhenApplied(const Hypothesis& hypo,
                           ScoreComponentCollection* accumulator) const {
  }
  void EvaluateWhenApplied(const ChartHypothesis &hypo,
                           ScoreComponentCollection* accumulator) const {
  }
  void EvaluateWhenApplied(const Syntax::SHyperedge &hyperedge,
                           ScoreComponentCollection* accumulator) const {
  }
  void EvaluateWithSourceContext(const InputType &input
                                 , const InputPath &inputPath
                                 , const TargetPhrase &targetPhrase
                                 , const StackVec *stackVec
                                 , ScoreComponentCollection &scoreBreakdown
                                 , ScoreComponentCollection *estimatedFutureScore = NULL) const {
  }

  void EvaluateTranslationOptionListWithSourceContext(const InputType &input
      , const TranslationOptionList &translationOptionList) const {
  }



  /*
    virtual void Evaluate(const InputType &source
                          , ScoreComponentCollection &scoreBreakdown) const;
  */
};

}


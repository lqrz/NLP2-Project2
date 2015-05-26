#include "Word2VecFeature.h"
#include "moses/TargetPhrase.h"
#include "moses/ScoreComponentCollection.h"
#include "util/tokenize.hh"

using namespace std;

template<typename T>
void OutputVec(const vector<T> &vec)
{
  for (size_t i = 0; i < vec.size(); ++i) {
    cerr << vec[i] << " " << flush;
  }
  cerr << endl;
}

namespace Moses
{
Word2VecFeature *Word2VecFeature::s_instance = NULL;

Word2VecFeature::Word2VecFeature(const std::string &line)
  : StatelessFeatureFunction(1, line)
{
  ReadParameters();

  UTIL_THROW_IF2(s_instance, "Can only have 1 word penalty feature");
  s_instance = this;
}


//void DecodeFeature::SetParameter(const std::string& key, const std::string& value)
void Word2VecFeature::SetParameter(const std::string& key, const std::string& value)
{
  string filePath;
  if (key == "path") {
  	m_tableStr.Tokenize(value, ",");
    //m_input =Tokenize<FactorType>(value, ",");
    ///m_inputFactors = FactorMask(m_input);
  //} else if (key == "output-factor") {
    //m_output =Tokenize<FactorType>(value, ",");
    //m_outputFactors = FactorMask(m_output);
  } else {
  	//call the parents method if param is unknown
    StatelessFeatureFunction::SetParameter(key, value);
  }
}

//void LanguageModelSRI::Load()
void Word2VecFeature::Load()
{
  string table = m_tableStr[i];
  lexicalTable* e2f = new lexicalTable;
  LoadLexicalTable(lex_e2f, e2f);

  //m_srilmVocab  = new ::Vocab();
  //m_srilmModel	= new Ngram(*m_srilmVocab, m_nGramOrder);

  //m_srilmModel->skipOOVs() = false;

  //File file( m_filePath.c_str(), "r" );
  //m_srilmModel->read(file);

  // LM can be ok, just outputs warnings
  //CreateFactors();
  //m_unknownId = m_srilmVocab->unkIndex();
}


void Word2VecFeature::LoadLexicalTable( string &fileName, lexicalTable* ltable)
{

  cerr << "Loading lexical translation table from " << fileName;
  ifstream inFile;
  inFile.open(fileName.c_str());
  if (inFile.fail()) {
    cerr << " - ERROR: could not open file\n";
    exit(1);
  }
  istream *inFileP = &inFile;

  int i=0;
  string line;

  while(getline(*inFileP, line)) {
    i++;
    if (i%100000 == 0) cerr << "." << flush;

    const vector<string> token = util::tokenize( line );
    if (token.size() != 4) {
      cerr << "line " << i << " in " << fileName
           << " has wrong number of tokens, skipping:\n"
           << token.size() << " " << token[0] << " " << line << endl;
      continue;
    }

    double joint = atof( token[2].c_str() );
    double marginal = atof( token[3].c_str() );
    Word wordT, wordS;
    wordT.CreateFromString(Output, m_output, token[0], false);
    wordS.CreateFromString(Input, m_input, token[1], false);
    ltable->joint[ wordS ][ wordT ] = joint;
    ltable->marginal[ wordS ] = marginal;
  }
  cerr << endl;

}

void Word2VecFeature::EvaluateInIsolation(const Phrase &source
    , const TargetPhrase &targetPhrase
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection &estimatedFutureScore) const
{
  float score = - (float) targetPhrase.GetNumTerminals();
  scoreBreakdown.Assign(this, score);
}

}


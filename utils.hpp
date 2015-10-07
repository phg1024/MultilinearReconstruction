#ifndef UTILS_H
#define UTILS_H

#include "common.h"
#include <typeinfo>
#include <limits>

inline void wait()
{
  std::cout << "Press ENTER to continue...";
  std::cin.ignore( std::numeric_limits <std::streamsize> ::max(), '\n' );
}

// misc
template <typename T>
ostream& printArray(T* A, int N, ostream& os = cout)
{
  for(int i=0;i<N;i++)
    os << A[i] << ' ';
  os << endl;
  return os;
}

template <typename T>
ostream& print2DArray(T** A, int rows, int cols, ostream& os = cout)
{
  for(int i=0;i<rows;i++)
  {
    for(int j=0;j<cols;j++)
    {
      os << A[i][j] << ' ';
    }
    os << endl;
  }
  return os;
}

template <typename T>
string toString(const T& val) {
  stringstream ss;
  ss << val;
  return ss.str();
}

// debugging related
// dummy
static void debug(){}

template <typename T>
void debug(const string& name, T value)
{
  cout << name << " = " << value << endl;
}

template <typename T, typename ...Args>
void debug(const string& name, T value, Args ...args)
{
  cout << name << " = " << value << endl;
  debug(args...);
}

// general console message
inline void message(const string& msg) {
  cout << msg << endl;
}

inline void error(const string& msg) {
  cerr << "Error:\t" << msg << endl;
}

inline void abort(const string& msg) {
  cerr << "Critical error:\t" << msg << endl;
  exit(0);
}

// exception related
template <typename T>
class ExceptionBase : public exception
{
public:
  ExceptionBase(const string& str = ""):exception(){
    if( !str.empty() )
      msg = string(typeid(T).name()) + " :: " + str;
    else
      msg = string(typeid(T).name());
  }

  virtual const char* what() const throw()
  {
    return msg.c_str();
  }

private:
  string msg;
};

struct LazyProgrammerException{};
typedef ExceptionBase<LazyProgrammerException> lazy_exception;

#endif // UTILS_H


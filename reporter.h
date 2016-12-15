#pragma once

#include "common.h"

#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;

class Reporter {
public:
  Reporter() {}
  Reporter(const string& name) : name(name) {}
  ~Reporter() {
    PrintReport();
  }

  Reporter& AddToEntry(const string& event, double time_cost);
  void PrintReport(ostream& os = cout) const;

  void Tic(const string& event);
  void Toc(const string& event = string(), ostream& os=cout);

  Reporter& operator<<(const pair<string, double>& event) {
    return AddToEntry(event.first, event.second);
  }

  ostream& operator<<(ostream& os) {
    PrintReport(os);
    return os;
  }

private:
  void PrintBanner(ostream& os) const {
    for(int i=0;i<80;++i) {
      os << '=';
    }
    os << "\n";
  }

private:
  static const int64_t one_second = 1000000000LL;
  string name;
  string last_event;
  map<string, double> timers;
  map<string, boost::timer::cpu_timer> clocks;
};

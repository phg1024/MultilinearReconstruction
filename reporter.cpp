#include "reporter.h"

Reporter& Reporter::AddToEntry(const string& event, double time_cost) {
  if(timers.count(event)) {
    timers[event] += time_cost;
  } else {
    timers.insert(make_pair(event, time_cost));
  }

  return (*this);
}

void Reporter::PrintReport(ostream& os) const {
  if(timers.empty()) return;

  PrintBanner(os);
  os << name << ": " << GetElapsed(self_timer) << " seconds.\n";
  PrintBanner(os);
  using record_t = pair<string, double>;
  vector<record_t> records(timers.begin(), timers.end());
  std::sort(records.begin(), records.end(),
            [](const record_t& a, const record_t& b) {
              return a.second > b.second;
            });
  for(auto p : records) {
    os << p.first << ": " << p.second << " seconds.\n";
  }
  PrintBanner(os);
  os << endl;
}

void Reporter::Tic(const string& event) {
  clocks[event] = cpu_timer();
  last_event = event;
}

void Reporter::Toc(const string& e, ostream& os) {
  string event = e;
  if(e.empty())  event = last_event;

  if(clocks.count(event)) {
    auto& timer = clocks[event];
    timer.stop();
    AddToEntry(event, GetElapsed(timer));
    os << event << ": " << timer.format() << endl;
    clocks.erase(event);
  }
}

double Reporter::GetElapsed(const cpu_timer& timer) const {
  cpu_times const elapsed_times(timer.elapsed());
  nanosecond_type const elapsed(elapsed_times.wall);
  return elapsed / static_cast<double>(one_second);
}

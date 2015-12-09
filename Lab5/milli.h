#ifndef _MILLI_
#define _MILLI_

#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> dsec;

class Timer {
private:
	Time::time_point startTime;
	double lapTime, time;

public:
	Timer();

	double GetMilliseconds();
	double GetMicroseconds();
	double GetSeconds();
	double GetTotalMilliseconds();
	double GetTotalMicroseconds();
	double GetTotalSeconds();
	void ResetMilli();
	void StartTimer();
	void EndTimer();
};

/*
int GetMilliseconds();
int GetMicroseconds();
double GetSeconds();

// Optional setting of the start time. If these are not used,
// the first call to the above functions will be the start time.
void ResetMilli();
void SetMilli(int seconds, int microseconds);
*/


#endif
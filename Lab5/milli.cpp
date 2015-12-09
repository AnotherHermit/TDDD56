// Simple little unit for timing using the gettimeofday() call.
// By Ingemar 2009

#include <stdlib.h>
#include "milli.h"


Timer::Timer() {
	time = 0.0f;
	lapTime = 0.0f;
}

void Timer::ResetMilli() {
	StartTimer();
}

void Timer::StartTimer() {
	startTime = Time::now();
}

void Timer::EndTimer() {
	double tempTime = dsec(Time::now() - startTime).count();
	lapTime = tempTime - time;
	time = tempTime;
}

double Timer::GetTotalSeconds() {
	return time;
}

double Timer::GetTotalMilliseconds() {
	return time * 1000.0f;
}

double Timer::GetTotalMicroseconds() {
	return time * 1000000.0f;
}

double Timer::GetSeconds() {
	return lapTime;
}

double Timer::GetMilliseconds() {
	return lapTime * 1000.0f;
}

double Timer::GetMicroseconds() {
	return lapTime * 1000000.0f;
}



/*

int GetMilliseconds() {
	timeval tv;

	gettimeofday(&tv, NULL);
	if (!hasStart) {
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) / 1000 + (tv.tv_sec - timeStart.tv_sec) * 1000;
}

int GetMicroseconds() {
	timeval tv;

	gettimeofday(&tv, NULL);
	if (!hasStart) {
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) + (tv.tv_sec - timeStart.tv_sec) * 1000000;
}

double GetSeconds() {
	timeval tv;

	gettimeofday(&tv, NULL);
	if (!hasStart) {
		hasStart = 1;
		timeStart = tv;
	}
	return (double)(tv.tv_usec - timeStart.tv_usec) / 1000000.0 + (double)(tv.tv_sec - timeStart.tv_sec);
}

// If you want to start from right now.
void ResetMilli() {
	timeval tv;

	gettimeofday(&tv, NULL);
	hasStart = 1;
	timeStart = tv;
}

// If you want to start from a specific time.
void SetMilli(int seconds, int microseconds) {
	hasStart = 1;
	timeStart.tv_sec = seconds;
	timeStart.tv_usec = microseconds;
}

*/
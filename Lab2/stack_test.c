/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

stack_t *stack;
data_t data;
pthread_barrier_t aba_barrier;

#if MEASURE >= 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
	stack_pop(stack);
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    {
        // See how fast your implementation can push MAX_PUSH_POP elements in parallel
        stack_push(stack,i);
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  // Initialize your test batch
}

void
test_setup()
{
  int i;
  stack_element_t* element;
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));
  stack->head = NULL;

  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):
  for (i = 0; i < MAX_PUSH_POP; i++){
    element = malloc(sizeof(stack_element_t));
    element->value = i;
    stack_push(stack,element);  
  }

}

void
test_teardown()
{
  stack_element_t* element;
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  while (stack->head != NULL) {
    element = stack_pop(stack);
    free(element);
  }

  free(stack);
}

void
test_finalize()
{
  // Destroy properly your test batch
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it
  int i;
  stack_element_t* element;  

  // Do some work
  for (i = 10; i < 20; i++) {
    element = malloc(sizeof(stack_element_t));
    element->value = i;
    stack_push(stack, element);
  }

  // check if the stack is in a consistent state
  stack_check(stack);

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  assert(stack->head->value == 19);

  // For now, this test always fails
  return 1;
}

int
test_pop_safe()
{
  int i;
  // Same as the test above for parallel pop operation
  for (i = MAX_PUSH_POP; i > 0; i--) {
    assert(stack_pop(stack)->value == i-1);
  }

  // For now, this test always fails
  return 1;
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

pthread_mutex_t aba_m_1;

void print_stack(const char* input){
  stack_element_t* el;
  el = stack->head;
  printf("\n%s, stack is now: ", input);
  while (el != NULL){
    printf("%c ", el->value);
    el = el->next;
  }
  printf("\n");
}

// A B C
void* aba_1(void* arg)
{
  pthread_mutex_lock(&aba_m_2);
  pthread_barrier_wait(&aba_barrier);
  // Pop A (CAS Head = A)
  // expected: B C
  stack_pop_aba(stack);   // -- PREEMPTED  
  // Result should be: A D B C 
  // But is: B C
  // D is gone :(
  print_stack("Finished");
  return NULL;
}

// A B C D
void* aba_2(void* arg)
{
  stack_element_t* element;
  stack_element_t* keeper;

  pthread_mutex_lock(&aba_m_1);
  pthread_barrier_wait(&aba_barrier);
  pthread_mutex_lock(&aba_m_2);

  // Pop and store A > B C
  keeper = stack_pop(stack);
  print_stack("Pop A");
  // Push D > D B C
  element = malloc(sizeof(stack_element_t));
  element->value = 'D';
  stack_push(stack, element);
  print_stack("Push D");
  // Push A > A D B C
  stack_push(stack, keeper);
  print_stack("Push A");
  // Control back to thread 1
  pthread_mutex_unlock(&aba_m_1);
  return NULL;
}

int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  // Write here a test for the ABA problem
  pthread_attr_t attr;
  pthread_t thread[2];
  stack_measure_arg_t args[2];

  stack_element_t* element;

  int i, success, aba_detected = 0;;
  while (stack->head != NULL){
    free(stack_pop(stack));
  }

  element = malloc(sizeof(stack_element_t));
  element->value = 'C';
  stack_push(stack, element);
  element = malloc(sizeof(stack_element_t));
  element->value = 'B';
  stack_push(stack, element);
  element = malloc(sizeof(stack_element_t));
  element->value = 'A';
  stack_push(stack, element);

  print_stack("Init");

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_barrier_init(&aba_barrier, NULL, 3);

  pthread_create(&thread[0], &attr, &aba_1, (void*) &args[0]);
  pthread_create(&thread[1], &attr, &aba_2, (void*) &args[1]);

  pthread_barrier_wait(&aba_barrier);

  for (i = 0; i < 2; i++)
  {
      pthread_join(thread[i], NULL);
  }


  success = aba_detected;
  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  assert(success);
  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];

  test_setup();
  pthread_attr_init(&attr);

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
#endif

  return 0;
}

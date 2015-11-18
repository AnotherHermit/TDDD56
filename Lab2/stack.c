/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif



void
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
}



void 
stack_push(stack_t* stack, stack_element_t* new_element)
{
  size_t cas_result;
  stack_element_t* expected;
  //printf("+ %d\n", value);

#if NON_BLOCKING == 0
  // Implement a lock_based stack  
  pthread_mutex_lock(&stack->mutex);  
  new_element->next = stack->head;
  stack->head = new_element;
  pthread_mutex_unlock(&stack->mutex);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  do {
    expected = stack->head;
    new_element->next = stack->head;
    cas_result = cas(
      (size_t*)&stack->head,
      (size_t)expected,
      (size_t)new_element);
    //printf("CAS failed\n");
  } while (cas_result != (size_t)expected);
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);
}

stack_element_t*
stack_pop(stack_t* stack)
{
  stack_element_t* head;
  stack_element_t* expected;
  size_t cas_result;

#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&stack->mutex);  
  head = stack->head;
  stack->head = head->next;
  pthread_mutex_unlock(&stack->mutex);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  do {
    expected = stack->head;
    head = stack->head;
    cas_result = cas(
      (size_t*)&stack->head,
      (size_t)expected,
      (size_t)head->next);
    //printf("CAS failed\n");
  } while (cas_result != (size_t)expected);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  //printf("- %d\n", value);
  return head;
}


stack_element_t*
stack_pop_aba(stack_t* stack)
{
  stack_element_t* head;
  stack_element_t* expected;
  size_t cas_result;

#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&stack->mutex);  
  head = stack->head;
  stack->head = head->next;
  pthread_mutex_unlock(&stack->mutex);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  do {
    expected = stack->head;
    head = stack->head;
    pthread_mutex_unlock(&aba_m_2);
    pthread_mutex_lock(&aba_m_1);
    pthread_mutex_unlock(&aba_m_1);
    printf("Thread 0 starting after lock1\n");
    cas_result = cas(
      (size_t*)&stack->head,
      (size_t)expected,
      (size_t)head->next);
    //printf("CAS failed\n");
  } while (cas_result != (size_t)expected);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  //printf("- %d\n", value);
  return head;
}



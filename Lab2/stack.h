/*
 * stack.h
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

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

struct stack_element;
pthread_mutex_t aba_m_1;
pthread_mutex_t aba_m_2;

typedef struct
{
  struct stack_element* head;
  pthread_mutex_t mutex;
} stack_t;

typedef struct stack_element
{
  int value;
  struct stack_element* next;
} stack_element_t;


// Pushes an element in a thread-safe manne
void 
stack_push(stack_t* stack, stack_element_t* element);

// Pops an element in a thread-safe manner
stack_element_t*
stack_pop(stack_t* stack);

// Pops an element in a thread-safe manner
stack_element_t*
stack_pop_aba(stack_t* stack);


/* Debug practice: check the boolean expression expr; if it computes to 0, print a warning message on standard error and exit */

// If a default assert is already defined, undefine it first
#ifdef assert
#undef assert
#endif

// Enable assert() only if NDEBUG is not set
#ifndef NDEBUG
#define assert(expr) if(!expr) { fprintf(stderr, "[%s:%s:%d][ERROR] Assertion failure: %s\n", __FILE__, __FUNCTION__, __LINE__, #expr); abort(); }
#else
// Otherwise define it as nothing
#define assert(expr)
#endif

// Debug practice: function that can check anytime is a stack is in a legal state using assert() internally
void
stack_check(stack_t *stack);

#endif /* STACK_H */

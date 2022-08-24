// Copyright (C) 2022 ATHENA DECODER AUTHORS; Rui Yan
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef THREADPOOL_H_
#define THREADPOOL_H_

#include <stdio.h>
#include <queue>
#include <unistd.h>
#include <pthread.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>

namespace athena {

class Task {
 public:
  Task() = default;
  virtual ~Task() = default;
  virtual void run()=0;
};

// Reusable thread class
class Thread {
 public:
  Thread() {
    state = EState_None;
    handle = 0;
  }

  virtual ~Thread() {
    assert(state != EState_Started);
  }

  void start() {
    assert(state == EState_None);
    // in case of thread create error I usually FatalExit...
    if (pthread_create(&handle, nullptr, threadProc, this))
      abort();
    state = EState_Started;
  }

  void join() {
    // A started thread must be joined exactly once!
    // This requirement could be eliminated with an alternative implementation but it isn't needed.
    assert(state == EState_Started);
    pthread_join(handle, nullptr);
    state = EState_Joined;
  }

 protected:
  virtual void run() = 0;

 private:
  static void* threadProc(void* param) {
    auto thread = reinterpret_cast<Thread*>(param);
    thread->run();
    return nullptr;
  }

 private:
  enum EState {
    EState_None,
    EState_Started,
    EState_Joined
  };

  EState state;
  pthread_t handle;
};

// Wrapper around std::queue with some mutex protection
class WorkQueue {
 public:
  WorkQueue() {
    pthread_mutex_init(&qmtx,nullptr);

    // wcond is a condition variable that's signaled
    // when new work arrives
    pthread_cond_init(&wcond, nullptr);
  }

  ~WorkQueue() {
    // Cleanup pthreads
    pthread_mutex_destroy(&qmtx);
    pthread_cond_destroy(&wcond);
  }

  // Retrieves the next task from the queue
  Task *nextTask() {
    // The return value
    Task *nt = nullptr;

    // Lock the queue mutex
    pthread_mutex_lock(&qmtx);

    while (tasks.empty())
      pthread_cond_wait(&wcond, &qmtx);

    nt = tasks.front();
    tasks.pop();

    // Unlock the mutex and return
    pthread_mutex_unlock(&qmtx);
    return nt;
  }
  // Add a task
  void addTask(Task *nt) {
    // Lock the queue
    pthread_mutex_lock(&qmtx);
    if (tasks.size() < 100000) {
      // Add the task
      tasks.push(nt);
    }
    // signal there's new work
    pthread_cond_signal(&wcond);
    // Unlock the mutex
    pthread_mutex_unlock(&qmtx);
  }

 private:
  std::queue<Task*> tasks;
  pthread_mutex_t qmtx;
  pthread_cond_t wcond;
};

// Thanks to the reusable thread class implementing threads is
// simple and free of pthread api usage.
class PoolWorkerThread : public Thread {
 public:
  explicit PoolWorkerThread(WorkQueue& _work_queue) : work_queue(_work_queue) {}
 protected:
  void run() override {
    while (Task* task = work_queue.nextTask()){
      task->run();
      delete task;
    }
  }
 private:
  WorkQueue& work_queue;
};

class ThreadPool {
 public:
  // Allocate a thread pool and set them to work trying to get tasks
  explicit ThreadPool(int n) {
    printf("Creating a thread pool with %d threads\n", n);
    for (int i=0; i<n; ++i) {
      threads.push_back(new PoolWorkerThread(workQueue));
      threads.back()->start();
    }
  }

  // Wait for the threads to finish, then delete them
  ~ThreadPool() {
    finish();
  }

  // Add a task
  void addTask(Task *nt) {
    workQueue.addTask(nt);
  }

  // Asking the threads to finish, waiting for the task
  // queue to be consumed and then returning.
  void finish() {
    for (size_t i=0,e=threads.size(); i<e; ++i)
      workQueue.addTask(nullptr);
    for (auto it : threads) {
      it->join();
      delete it;
    }
    threads.clear();
  }

 private:
  std::vector<PoolWorkerThread*> threads;
  WorkQueue workQueue;
};

} // namespace athena

#endif//THREADPOOL_H_
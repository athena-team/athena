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

#include <string>
#include <thread>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <hv/EventLoop.h>
#include <hv/WebSocketServer.h>

#include "server/server.h"

using namespace hv;

DEFINE_int32(port, 8888, "listening port");
DEFINE_int32(threads, 10, "listening threads");

int OnHttpPostMessage(const HttpContextPtr& ctx) {
  auto reqbody = ctx->body();
  LOG(INFO) << "OnWSocketOpen. peerAddress=" << ctx->host() << " msg=" << reqbody;
  return ctx->send(reqbody);
}

void OnWSocketOpen(const WebSocketChannelPtr& channel, const std::string& url) {
  channel->newContext< athena::SessionHandler >();
  LOG(INFO) << "OnWSocketOpen. peerAddress=" << channel->peeraddr() << " url=" << url;
}

void OnWSocketMessage(const WebSocketChannelPtr& channel, const std::string& msg) {
  auto ctx      = channel->getContext< athena::SessionHandler >();
  auto response = ctx->OnReceiveMessage(msg);
  channel->send(response);
  LOG(INFO) << "OnWSocketMessage send: response==" << response;
}

void OnWSocketClose(const WebSocketChannelPtr& channel) {
  auto ctx = channel->getContext< athena::SessionHandler >();
  LOG(INFO) << "OnWSocketClose. peerAddress=" << channel->peeraddr() << " traceid=" << ctx->GetTraceid();
  ctx->OnSessionClose();
  channel->deleteContext< athena::SessionHandler >();
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  if (!InitGlobalResource(athena::kEnd2End)) {
    LOG(INFO) << "load decoder resource failed";
    exit(-1);
  }
  LOG(INFO) << "load decoder resource finished";

  HttpService http;
  http.GET("/ping", OnHttpPostMessage);
  http.POST("/ping", OnHttpPostMessage);
  WebSocketService ws;
  ws.onopen    = OnWSocketOpen;
  ws.onmessage = OnWSocketMessage;
  ws.onclose   = OnWSocketClose;

  websocket_server_t server;
  server.port           = FLAGS_port;
  server.service        = &http;
  server.ws             = &ws;
  server.worker_threads = FLAGS_threads;

  LOG(INFO) << "Listening at port: " << FLAGS_port << "threads: " << FLAGS_threads;
  websocket_server_run(&server, 0);
  while (getchar() != '\n');
  http_server_stop(&server);
  LOG(INFO) << "server stoping";
  return 0;
}

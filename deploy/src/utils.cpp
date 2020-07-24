/* Copyright (C) 2020 ATHENA AUTHORS;
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================== */

#include <string>
#include <fstream>
#include <vector>
#include "../include/utils.h"


void createMap(std::vector<std::string> &index_to_char, std::string filename) {
    std::ifstream fin(filename);
    std::string str = "";
    int index = 0;
    while(fin >> str >> index) {
        if (str.compare("<space>") == 0) {
            str = " ";
        }
        index_to_char.push_back(str);
    }
    fin.close();
}

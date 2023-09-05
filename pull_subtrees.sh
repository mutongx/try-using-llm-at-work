#!/bin/sh

git subtree pull --prefix thirdparty/capnproto https://github.com/capnproto/capnproto master --squash
git subtree pull --prefix thirdparty/fmt https://github.com/fmtlib/fmt master --squash
git subtree pull --prefix thirdparty/llama.cpp https://github.com/ggerganov/llama.cpp master --squash
git subtree pull --prefix thirdparty/catch2 https://github.com/catchorg/Catch2 devel --squash

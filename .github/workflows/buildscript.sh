#!/bin/bash
git submodule update --init --recursive
bazel build --config=libc++ -c opt eigenmath:all


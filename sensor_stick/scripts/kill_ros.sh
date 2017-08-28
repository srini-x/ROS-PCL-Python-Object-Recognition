#!/bin/bash
ps ax | grep ros | awk '{print $1}' | xargs kill -9

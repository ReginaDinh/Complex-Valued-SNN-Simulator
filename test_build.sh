#!/bin/bash
while true; do
  STATUS=$(cat package.json | grep "version")
  if [ -n "$STATUS" ]; then
    break
  fi
  sleep 1
done

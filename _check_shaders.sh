#!/bin/bash
# Temporary: compile-check the served shader bundle with glslang.
cd /tmp || exit 1
for f in height.frag display.frag vertex.vert; do
  /tmp/glslang/bin/glslang "/tmp/$f" > "/tmp/log_$f.txt" 2>&1
  echo "$f exit=$? lines=$(wc -l < /tmp/log_$f.txt)"
done
echo "---- errors/warnings:"
grep -iE "error|warn" /tmp/log_*.txt | head -20
echo "---- done"

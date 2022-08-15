#!/usr/bin/env bash
status=0
# check some system-level installations you need to do in this script
# linux
if which yum >&/dev/null; then
   echo "$0: you system is Centos, you should change the following apt or apt-get to yum. but we recommend using ubuntu !"
  status=1
  exit 1;
fi

if ! which apt-get >&/dev/null || ! which apt >/dev/null; then
  echo "$0: We just provide the installation steps with the corresponding linux system environment."
  echo "$0: We suggest you switch to Ubuntu."
  status=1
  exit 1;
fi


# which
if ! which which >&/dev/null; then
  echo "$0: which is not installed."
  status=1
  exit 1;
fi

# tensorflow-text

ver=`pip list | grep -v tensorflow- |grep  tensorflow | awk '{print $2}'`
tftext="tensorflow-text==$ver"

if [ $ver = "2.3.1" ]; then
     echo "tensorflow-text==2.3.0" >> requirements.txt
else
     echo $tftext >> requirements.txt
fi

# python3
if ! which python3 >&/dev/null; then
  echo "$0: python3 is not installed"
  echo "$0: This project has only been tested on Python 3, we recommend using Python 3"
  status=1
  exit 1;
fi

# wget git sox cmake
for f in wget git sox cmake; do
  if ! which $f >&/dev/null; then
    echo "$0: $f is not installed."
    echo "You should probably do: "
    echo " sudo apt-get update && sudo apt-get install $f"
	status=1
  fi
done

if [ $status -eq 0 ]; then
  echo "$0: system-level installations you need to do all OK."
fi

exit $status

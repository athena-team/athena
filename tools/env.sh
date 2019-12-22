# check if we are executing or sourcing.
if [[ "$0" == "$BASH_SOURCE" ]]
then
    echo "You must source this script rather than executing it."
    exit -1
fi

# don't use readlink because macOS won't support it.
if [[ "$BASH_SOURCE" == "/"* ]]
then
    export MAIN_ROOT=$(dirname $BASH_SOURCE)
else
    export MAIN_ROOT=$PWD/$(dirname $BASH_SOURCE)
fi

# pip bins
export PATH=$PATH:~/.local/bin

# athena
export PYTHONPATH=${PYTHONPATH}:$MAIN_ROOT:

#!/usr/bin/env bash

# install latest cmdstan release into subdir "releases"
# symlink release dir as "cmdstan"
# build binaries, compile example model to build model header

if [[ ! -e releases ]]; then
   mkdir releases
fi
if [[ ! -d releases ]]; then
    echo 'cannot install cmdstan, file "releases" is not a directory'
    exit 1
fi
pushd releases

echo `pwd`

TAG=`curl -s https://api.github.com/repos/stan-dev/cmdstan/releases/latest | grep "tag_name"`
echo $TAG > tmp-tag
VER=`perl -p -e 's/"tag_name": "v//g; s/",//g' tmp-tag`
echo $VER

# VER=`curl -s https://api.github.com/repos/stan-dev/cmdstan/releases/latest | grep "tag_name" | sed -E 's/.*"v([^"]+)".*/\1/'`

cs=cmdstan-${VER}
if [[ -d $cs && -f $cs/bin/stanc && -f $cs/examples/bernoulli/bernoulli ]]; then
    echo "cmdstan already installed"
    exit 0
fi

curl -OL https://github.com/stan-dev/cmdstan/releases/download/v${VER}/${cs}.tar.gz

tar xzf ${cs}.tar.gz
if [[ -h cmdstan ]]; then
    unlink cmdstan
fi
ln -s ${cs} cmdstan
cd cmdstan
make -j2 build examples/bernoulli/bernoulli
echo "installed $cs"
echo `ls -lFd releases/*`

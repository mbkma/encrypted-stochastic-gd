# Encrypted Stochastic Gradient Descent

This is the encrypted stochastic gradient descent algorithm using secure two-party computation and the ABY framework.

### Requirements
---

* A **Linux distribution** of your choice (ESGD was developed and tested with recent versions of [Debian](https://www.debian.org/).
* **Required packages for ABY:**
  * [`g++`](https://packages.debian.org/testing/g++) (version >=8)
    or another compiler and standard library implementing C++17 including the filesystem library
  * [`make`](https://packages.debian.org/testing/make)
  * [`cmake`](https://packages.debian.org/testing/cmake)
  * [`libgmp-dev`](https://packages.debian.org/testing/libgmp-dev)
  * [`libssl-dev`](https://packages.debian.org/testing/libssl-dev)
  * [`libboost-all-dev`](https://packages.debian.org/testing/libboost-all-dev) (version >= 1.66)

  Install these packages with your favorite package manager, e.g, `sudo apt install <package-name>`.

#### Building ESGD

1. Clone the git repository by running:
    ```
    git clone https://github.com/mbkma/encrypted-stochastic-gd.git
    ```
2. Init git submodules:
    ```
    git submodule update --init --recursive
    ```
3. Use CMake configure the build:
    ```
    mkdir build && cd build && cmake ..
    ```
4. Call `make` in the build directory.
    ```
    make -j5
    ```

#### Example

    ```
    ./main -r 0 -l 0.03 -a 192.168.10
    ```

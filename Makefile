CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -march=native -Iinclude
LDFLAGS = 

SRCS = src/main.cpp \
       src/model/gguf_parser.cpp \
       src/core/compute.cpp \
       src/layers/layers.cpp \
       src/engine.cpp \
       src/server.cpp

OBJS = $(SRCS:.cpp=.o)
TARGET = forge_cli
LDFLAGS = -pthread
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

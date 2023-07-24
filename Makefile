CC = g++
CFLAGS = -Wall -Wextra -pedantic -std=c++17 -I/usr/include/python3.8
LDFLAGS = -lpython3.8 -ldl

# Name of the executable file
TARGET = ppo_training

# Source file
SRCS = src/main.cpp src/PythonWrapper.* src/AgentWrapper.* 

# Object files
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

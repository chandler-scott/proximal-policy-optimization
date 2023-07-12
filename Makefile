CC = g++
CFLAGS = -Wall -Wextra -pedantic -std=c++11 -I/usr/include/python3.8
LDFLAGS = -lpython3.8

# Name of the executable file
TARGET = ppo_training

# Source file
SRCS = src/main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

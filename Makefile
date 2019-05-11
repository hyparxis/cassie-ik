MJ_PATH := $(HOME)/.mujoco/mujoco200
INC		:= -Iinclude -I$(MJ_PATH)/include -L$(MJ_PATH)/bin $(MJ_PATH)/bin/libglfw.so.3
CFLAGS	:= -Wall -Wextra -O3 -std=c++11 -pthread
LDFLAGS	:= -shared -Lsrc

CC 		:= g++
OUT		:= sim
LIBS	:= -lmujoco200 -lGL -lglew 


all:
	$(CC) src/$(OUT).cpp $(INC) $(CFLAGS) -o $(OUT) $(LIBS)

clean:
	rm -f $(OUT)


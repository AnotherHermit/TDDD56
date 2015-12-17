CPP_FILES := $(wildcard *.cpp)

OBJ_FILES_CPP := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

MAIN = find_max.cu
OUT = find_max

GCC = /usr/local/cuda/bin/nvcc -std=c++11

LINKS += -lGL -lGLU -lglut -lm -lXt -lX11 -L /usr/local/cuda/lib64 -lcudart

all: $(OBJ_FILES_COMMON) $(OBJ_FILES_CPP)
	$(GCC) obj/*.o $(MAIN) -o $(OUT) $(INC) $(LINKS)


obj/%.o : %.cpp
	$(GCC) -c -o $@ $< $(INC)

clean :
	rm obj/*.o


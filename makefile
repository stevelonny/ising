run: run.cpp ising.hpp
	g++ -o run.x run.cpp
test: test.cpp ising.hpp
	g++ -o test.x test.cpp
clean:
	rm -f *.x
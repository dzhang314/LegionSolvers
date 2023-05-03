CPPFLAGS=-std=c++17 -O3

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CLEANFILES=main.o PetscSolverBenchmark

main: main.cpp
	$(PETSC_CXXCOMPILE_SINGLE) -c main.cpp -o main.o
	${CLINKER} main.o ${PETSC_KSP_LIB} -o PETScSolverBenchmark

# make: certsolve and the CPU tools into bin/;  make gpu: the Metal (GPU) tools (macOS, Apple silicon)
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++20 -march=native -pthread -Wall
BIN = bin
BROB = $(addprefix solver/brob/,xcube.cpp facelets.cpp util.cpp cornersolver.cpp wingsolver.cpp)
CPU = certcheck axcheck diagcheck midcheck wingcheck pairdepth boundcheck axbcheck boundx axbx axbound opbound \
      certextend certimprove certmerge colconcise cornerbfs
GPU = gpubound gpuaxbcheck gpuboundx gpuaxbx

all: $(BIN)/certsolve $(addprefix $(BIN)/,$(CPU))
gpu: $(addprefix $(BIN)/,$(GPU))

$(BIN)/certsolve: solver/certsolve.cpp solver/certrest.h solver/certcache.h solver/certcube.h solver/sim/cube.h $(BROB)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -Isolver -Isolver/brob -Isolver/sim -o $@ solver/certsolve.cpp $(BROB)
$(BIN)/opbound: tools/opbound.cpp tools/axbound.cpp solver/certcube.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -Isolver -o $@ tools/opbound.cpp
$(BIN)/colconcise: tools/colconcise.cpp solver/certrest.h solver/certcube.h $(BROB)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -Isolver -Isolver/brob -o $@ tools/colconcise.cpp $(BROB)
$(BIN)/%: tools/%.cpp solver/certcube.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -Isolver -o $@ $<
$(BIN)/%: tools/gpu/%.mm solver/certcube.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -fobjc-arc -Isolver -o $@ $< -framework Metal -framework Foundation

# an end-to-end boundary run (GPU): UD 34, AX 42 and AX 41 from the certificate, search and check
boundary-demo: all gpu
	./boundary-demo.sh work

clean:
	rm -rf $(BIN)

.PHONY: all gpu boundary-demo clean

# kci_vec_pg — PGXS build. Needs pg_config (postgresql-server-dev-NN).
#   make && sudo make install      # then: CREATE EXTENSION kci_vec_pg;
# kci_vec.c is compiled from the repository root with -O3 -march=native. Set KV_MARCH to target a
# different machine (e.g. KV_MARCH=x86-64-v3 for any AVX2 CPU).
MODULE_big = kci_vec_pg
OBJS = kci_vec_pg.o kci_vec.o
EXTENSION = kci_vec_pg
DATA = kci_vec_pg--0.3.sql kci_vec_pg--0.1--0.2.sql kci_vec_pg--0.2--0.3.sql
PG_CPPFLAGS = -I$(srcdir)/..
SHLIB_LINK = -lpthread -lm
KV_MARCH ?= native

override with_llvm = no   # no LLVM bitcode for the JIT: kci_vec.o is built outside the PGXS rules

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

kci_vec.o: ../kci_vec.c ../kci_vec.h ../kci_crypt.h ../kci_win_compat.h
	$(CC) $(filter-out -D_GNU_SOURCE,$(CPPFLAGS)) -O3 -march=$(KV_MARCH) -fPIC -Wall -Wno-unused-function -Wno-misleading-indentation -Wno-maybe-uninitialized -c -o $@ $<


.PHONY: server client

server: 
	@$(MAKE) -C server run

client:
	@$(MAKE) -C client run

clean:
	@$(MAKE) -C server clean
	@$(MAKE) -C client clean

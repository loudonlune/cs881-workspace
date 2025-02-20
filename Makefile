
CHECKTHAT_TRACK?=https://gitlab.com/checkthat_lab/clef2025-checkthat-lab.git
RUN_PROFILE?=

.PHONY: noop run-all run-profile

noop:
	echo "Running nothing."

checkthat_data:
	git clone --depth 1 $(CHECKTHAT_TRACK) checkthat_data

run-profile: checkthat_data
	./task2tool checkthat-task2 together-ai $(RUN_PROFILE)

run-all: checkthat_data
	./task2tool checkthat-task2 together-ai test-zero-shot
	./task2tool checkthat-task2 together-ai test-one-shot

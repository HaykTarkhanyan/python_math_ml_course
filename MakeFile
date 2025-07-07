.PHONY: render push

# Use msg from command-line, default to "render"
msg ?= rendering

render:
	@echo ">> Running quarto render..."
	@quarto render
	@echo ">> Staging changes..."
	@git add .
	@echo ">> Committing with message: '$(msg)'"
	@git commit -m "$(msg)"
	@echo ">> Pushing to remote..."
	@git push
	@echo ">> Done"

push:
	@echo ">> won't run `quarto render`..."
	@echo ">> Staging changes..."
	@git add .
	@echo ">> Committing with message: '$(msg)'"
	@git commit -m "$(msg)"
	@echo ">> Pushing to remote..."
	@git push
	@echo ">> Done"
# https://github.com/quarto-dev/quarto-cli/discussions/3674
library(dplyr)
library(stringr)

print("Starting")

files <- system("git diff main origin/main --name-only -- . :^docs", intern=T) %>% 
    tibble(files=.) %>% 
    filter(str_detect(files, 'ipynb$')) %>% 
    pull(files)

print(paste0("Found ", length(files), " changed files."))

if(length(files) > 0){
  cat(
    paste0('Rendering uncommitted *ipynb files:\n\t', 
           paste0(files, collapse ="\n\t")
    ), 
  "\n")
  for(f in files){
    print(f)
    cmd <- paste0("quarto render ", f)
    print(cmd)
    system(cmd)
  }
}
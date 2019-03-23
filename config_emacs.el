;; Keeps things simple in setting up my local emacs environment for this repo.

(setq flycheck-clang-include-path (list (expand-file-name "~/git-working/Radio-Gridders/src/common")))
(setq flycheck-clang-language-standard "gnu++11")
(setq flycheck-clang-standard-library "libc++")
(setq flycheck-cuda-include-path (list (expand-file-name "~/git-working/Radio-Gridders/src/common")
				       (expand-file-name "~/git-working/Radio-Gridders/src/wstack")))

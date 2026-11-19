# ManiSkill (SAPIEN) environment for WSL2 + llvmpipe (CPU Vulkan render)
# source this before running .venv-maniskill scripts
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
# prevents sapien._vulkan_tricks from injecting its broken nvidia_icd.json
export SAPIEN_VULKAN_LIBRARY_PATH=""

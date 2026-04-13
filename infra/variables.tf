variable "hcloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_private_key_path" {
  description = "Path to SSH private key (for provisioners)"
  type        = string
  default     = "~/.ssh/id_ed25519"
}

variable "server_type" {
  description = "Hetzner server type (dedicated CPU)"
  type        = string
  default     = "ccx33"
}

variable "location" {
  description = "Hetzner datacenter location"
  type        = string
  default     = "nbg1"
}

variable "volume_size" {
  description = "Persistent volume size in GB"
  type        = number
  default     = 100
}

variable "git_branch" {
  description = "Git branch to checkout on the server"
  type        = string
  default     = "master"
}

terraform {
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.45"
    }
  }
  required_version = ">= 1.5"
}

provider "hcloud" {
  token = var.hcloud_token
}

# --- SSH Key ---

resource "hcloud_ssh_key" "bot" {
  name       = "ict-trading-bot"
  public_key = file(pathexpand(var.ssh_public_key_path))
}

# --- Persistent Volume (survives server destruction) ---

resource "hcloud_volume" "data" {
  name     = "ict-bot-data"
  size     = var.volume_size
  location = var.location
  format   = "ext4"

  lifecycle {
    # Re-enable after first successful seed-db
    prevent_destroy = false
  }
}

# --- Firewall ---

resource "hcloud_firewall" "bot" {
  name = "ict-bot-fw"

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction       = "out"
    protocol        = "tcp"
    port            = "1-65535"
    destination_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction       = "out"
    protocol        = "udp"
    port            = "1-65535"
    destination_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction       = "out"
    protocol        = "icmp"
    destination_ips = ["0.0.0.0/0", "::/0"]
  }
}

# --- Server (ephemeral — create for run, destroy after) ---

resource "hcloud_server" "bot" {
  name        = "ict-bot"
  server_type = var.server_type
  location    = var.location
  image       = "ubuntu-24.04"
  ssh_keys    = [hcloud_ssh_key.bot.id]
  firewall_ids = [hcloud_firewall.bot.id]

  user_data = templatefile("${path.module}/cloud-init.yml.tpl", {
    volume_id  = hcloud_volume.data.id
    git_branch = var.git_branch
  })

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }

  lifecycle {
    # Server is ephemeral — recreate on any config change
    create_before_destroy = false
  }
}

resource "hcloud_volume_attachment" "data" {
  volume_id = hcloud_volume.data.id
  server_id = hcloud_server.bot.id
  automount = false
}

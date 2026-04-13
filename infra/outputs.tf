output "server_ip" {
  description = "Public IPv4 of the bot server"
  value       = hcloud_server.bot.ipv4_address
}

output "volume_id" {
  description = "Persistent volume ID"
  value       = hcloud_volume.data.id
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh root@${hcloud_server.bot.ipv4_address}"
}

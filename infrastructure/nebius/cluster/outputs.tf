output "master_public_ip" {
  description = "The public IP address of the master training VM"
  value = trimsuffix(nebius_compute_v1_instance.training_h100_master.status.network_interfaces[0].public_ip_address.address, "/32")
}

output "master_internal_ip" {
  description = "The internal IP address of the master training VM"
  value = trimsuffix(nebius_compute_v1_instance.training_h100_master.status.network_interfaces[0].ip_address.address, "/32")
}

output "worker_public_ips" {
  description = "The public IP addresses of the worker training VMs"
  value = [for instance in nebius_compute_v1_instance.training_h100_workers :
    trimsuffix(instance.status.network_interfaces[0].public_ip_address.address, "/32")
  ]
}

output "worker_internal_ips" {
  description = "The internal IP addresses of the worker training VMs"
  value = [for instance in nebius_compute_v1_instance.training_h100_workers :
    trimsuffix(instance.status.network_interfaces[0].ip_address.address, "/32")
  ]
}

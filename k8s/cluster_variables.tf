variable "cpu_instance_type" {
    description = "Instance type for the head node"
    default = "t3a.xlarge"
}
variable "cpu_desired_capacity" {
    default = 1
}
variable "cpu_min_size" {
    default = 1
}
variable "cpu_max_size" {
    default = 2
}
variable "gpu_instance_type" {
    description = "Instance type for the worker node"
    default = "g5.xlarge"
}
variable "gpu_desired_capacity" {
    default = 0
}
variable "gpu_min_size" {
    default = 0
}
variable "gpu_max_size" {
    default = 2
}
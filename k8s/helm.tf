resource "helm_release" "ray_cluster" {
  name       = "ray"
  chart      = "./raycluster"  # path to your Helm chart
  namespace  = kubernetes_namespace.protoplast.metadata[0].name

  values = [
    yamlencode({
        awsRegion    = var.aws_region
        awsAccountId = var.aws_account_id
        s3Bucket     = aws_s3_bucket.protoplast_bucket
        cpuNodeGroup = {
        instanceType    = var.cpu_instance_type
        desiredCapacity = var.cpu_desired_capacity
        minSize         = var.cpu_min_size
        maxSize         = var.cpu_max_size
        }
        gpuNodeGroup = {
        instanceType    = var.gpu_instance_type
        desiredCapacity = var.gpu_desired_capacity
        minSize         = var.gpu_min_size
        maxSize         = var.gpu_max_size
        }
    })
  ]
}

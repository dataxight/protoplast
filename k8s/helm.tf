resource "helm_release" "kuberay_operator" {
  name       = "kuberay-operator"
  repository = "https://ray-project.github.io/kuberay-helm/"
  chart      = "kuberay-operator"
  version    = "1.4.2"
  namespace  = kubernetes_namespace.protoplast.metadata[0].name
  create_namespace = false
  wait = false
}


resource "helm_release" "ray_cluster" {
  name       = "ray"
  chart      = "./raycluster"
  namespace  = kubernetes_namespace.protoplast.metadata[0].name
  depends_on = [ helm_release.kuberay_operator ]

  values = [
    yamlencode({
        awsRegion    = var.aws_region
        awsAccountId = var.aws_account_id
        s3Bucket     = aws_s3_bucket.protoplast_bucket.bucket
        headServiceAccount = kubernetes_service_account.protoplast_head_sa.metadata[0].name
        workerServiceAccount = kubernetes_service_account.protoplast_worker_sa.metadata[0].name
        namespace = kubernetes_namespace.protoplast.metadata[0].name
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

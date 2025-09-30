data "aws_iam_policy_document" "trust" {
  statement {
    effect = "Allow"

    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    actions = ["sts:AssumeRoleWithWebIdentity"]

    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.oidc_provider, "https://", "")}:sub"
      values   = ["system:serviceaccount:protoplast:${kubernetes_namespace.protoplast.metadata[0].name}"]
    }
  }
}

resource "kubernetes_service_account" "protoplast_head_sa" {
    metadata {
        name = "protoplast-head-iam-${var.env}"
        namespace = kubernetes_namespace.protoplast.metadata[0].name
        annotations = {
            "eks.amazonaws.com/role-arn" = aws_iam_role.protoplast_head_role.arn
        }
    }
  
}

resource "kubernetes_service_account" "protoplast_worker_sa" {
    metadata {
        name = "protoplast-worker-iam-${var.env}"
        namespace = kubernetes_namespace.protoplast.metadata[0].name
        annotations = {
            "eks.amazonaws.com/role-arn" = aws_iam_role.protoplast_worker_role.arn
        }
    } 
}


resource "aws_iam_role" "protoplast_head_role" {
  name = "protoplast-protoplast-head-role-${var.env}"
  assume_role_policy = data.aws_iam_policy_document.trust.json
}

resource "aws_iam_role" "protoplast_worker_role" {
    name = "protoplast-protoplast-worker-role-${var.env}"
    assume_role_policy = data.aws_iam_policy_document.trust.json
}


data "aws_iam_policy_document" "ecr_access" {
    statement {
        actions = [
            "ecr:GetAuthorizationToken",
            "ecr:BatchCheckLayerAvailability",
            "ecr:GetDownloadUrlForLayer",
            "ecr:BatchGetImage"
        ]
        resources = ["*"]
    }
}

resource "aws_iam_role_policy" "head_s3_access" {
    name = "head_s3_access"
    role = aws_iam_role.protoplast_head_role.id
    policy = data.aws_iam_policy_document.protoplast_s3_access.json
}

resource "aws_iam_role_policy" "head_ecr_access" {
    name = "head_ecr_access"
    role = aws_iam_role.protoplast_head_role.id
    policy = data.aws_iam_policy_document.ecr_access.json
  
}

resource "aws_iam_role_policy" "worker_s3_access" {
    name = "worker_s3_access"
    role = aws_iam_role.protoplast_worker_role.id
    policy = data.aws_iam_policy_document.protoplast_s3_access.json
}


resource "aws_iam_role_policy" "worker_ecr_access" {
    name = "worker_ecr_access"
    role = aws_iam_role.protoplast_worker_role.id
    policy = data.aws_iam_policy_document.ecr_access.json
}

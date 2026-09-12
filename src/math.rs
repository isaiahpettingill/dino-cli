use crate::cli::Pool;
use anyhow::{bail, ensure, Result};
pub fn pool(data: &[f32], shape: &[i64], mode: Pool, skip: usize) -> Result<Vec<f32>> {
    ensure!(
        !data.is_empty() && data.iter().all(|x| x.is_finite()),
        "Empty or nonfinite output"
    );
    if matches!(mode, Pool::Flatten) {
        return Ok(data.to_vec());
    }
    match shape {
        [1, _] if matches!(mode, Pool::Auto) => Ok(data.to_vec()),
        [1, n, d] if matches!(mode, Pool::Cls | Pool::Mean) => {
            let (n,d)=(*n as usize,*d as usize);
            if matches!(mode, Pool::Cls) { return Ok(data[..d].to_vec()); }
            ensure!(skip<n, "skip-tokens must be less than token count");
            let mut out=vec![0.;d];
            for row in data[skip*d..].chunks_exact(d) { for (o,x) in out.iter_mut().zip(row) { *o += x / (n-skip) as f32; } }
            Ok(out)
        },
        [1,c,h,w] if matches!(mode,Pool::Spatial) => Ok(data.chunks_exact((h*w) as usize).take(*c as usize).map(|s| s.iter().sum::<f32>()/s.len() as f32).collect()),
        _ => bail!("Output shape {shape:?} needs explicit --pooling cls, mean, spatial (NCHW), or flatten; auto accepts only [1,D]"),
    }
}
pub fn normalize(v: &mut [f32]) -> Result<()> {
    let norm = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    ensure!(
        norm.is_finite() && norm > 0.,
        "Cannot normalize zero or nonfinite vector"
    );
    for x in v {
        *x = (*x as f64 / norm) as f32;
    }
    Ok(())
}
pub fn cosine(a: &[f32], b: &[f32]) -> Result<f64> {
    ensure!(
        a.len() == b.len() && !a.is_empty(),
        "Embedding dimensions differ or are empty"
    );
    let aa = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    let bb = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    ensure!(
        aa > 0. && bb > 0. && aa.is_finite() && bb.is_finite(),
        "Invalid embedding norm"
    );
    Ok((a
        .iter()
        .zip(b)
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum::<f64>()
        / aa.sqrt()
        / bb.sqrt())
    .clamp(-1., 1.))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn similarity() {
        assert!((cosine(&[3., 4.], &[6., 8.]).unwrap() - 1.).abs() < 1e-9);
        assert_eq!(cosine(&[1., 0.], &[0., 1.]).unwrap(), 0.);
        assert!(cosine(&[0.], &[1.]).is_err());
    }
    #[test]
    fn tokens() {
        assert_eq!(
            pool(&[9., 9., 1., 3., 3., 5.], &[1, 3, 2], Pool::Mean, 1).unwrap(),
            vec![2., 4.]
        );
        assert!(pool(&[1., 2.], &[1, 1, 2], Pool::Auto, 0).is_err());
    }
}

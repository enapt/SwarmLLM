//! QUIC pacing probe (FUTURE_WORK #318; `docs/DIAGNOSTICS.md` § "Is the transport
//! pacing a split's bursts?"). Two quinn endpoints on loopback; every <gap_ms> the
//! client opens a NEW bidirectional stream — as libp2p request-response opens
//! one per request — sends <size> bytes and waits for a 64-byte answer. Prints
//! the median send-to-answer time and the connection's final window.
//!
//! Args: <size> <gap_ms> <n> <prompt_bytes> <initial_window|0>; `CC=bbr` runs
//! quinn's BBR (what our QUIC connections run), anything else Cubic. Run it
//! inside `unshare -rn` with netem on `lo` (`run.sh`).
use std::{sync::Arc, time::{Duration, Instant}};

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let size: usize = a[1].parse().unwrap();
    let gap = Duration::from_millis(a[2].parse().unwrap());
    let n: usize = a[3].parse().unwrap();
    let prompt: usize = a[4].parse().unwrap();
    let iw: u64 = a[5].parse().unwrap();
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let der = rustls::pki_types::CertificateDer::from(cert.cert.der().to_vec());
    let key = rustls::pki_types::PrivateKeyDer::try_from(cert.key_pair.serialize_der()).unwrap();
    let mut transport = quinn::TransportConfig::default();
    // libp2p-quic's own windows
    transport.stream_receive_window(10_000_000u32.into());
    transport.receive_window(15_000_000u32.into());
    let bbr = std::env::var("CC").as_deref() == Ok("bbr");
    if bbr {
        let mut cc = quinn::congestion::BbrConfig::default();
        if iw > 0 { cc.initial_window(iw); }
        transport.congestion_controller_factory(Arc::new(cc));
    } else if iw > 0 {
        let mut cc = quinn::congestion::CubicConfig::default();
        cc.initial_window(iw);
        transport.congestion_controller_factory(Arc::new(cc));
    }
    let transport = Arc::new(transport);
    let mut sc = quinn::ServerConfig::with_single_cert(vec![der.clone()], key).unwrap();
    sc.transport = transport.clone();
    let server = quinn::Endpoint::server(sc, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = server.local_addr().unwrap();
    tokio::spawn(async move {
        let conn = server.accept().await.unwrap().await.unwrap();
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            tokio::spawn(async move {
                let _ = recv.read_to_end(usize::MAX).await;
                let _ = send.write_all(&[0u8; 64]).await;
                let _ = send.finish();
            });
        }
    });
    let mut roots = rustls::RootCertStore::empty();
    roots.add(der).unwrap();
    let mut cc = quinn::ClientConfig::with_root_certificates(Arc::new(roots)).unwrap();
    cc.transport_config(transport);
    let mut client = quinn::Endpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
    client.set_default_client_config(cc);
    let conn = client.connect(addr, "localhost").unwrap().await.unwrap();
    let one = |bytes: usize| {
        let conn = conn.clone();
        async move {
            let t = Instant::now();
            let (mut s, mut r) = conn.open_bi().await.unwrap();
            s.write_all(&vec![1u8; bytes]).await.unwrap();
            s.finish().unwrap();
            let _ = r.read_to_end(1024).await.unwrap();
            t.elapsed().as_secs_f64() * 1000.0
        }
    };
    if prompt > 0 {
        let p = one(prompt).await;
        println!("prompt {prompt} B took {p:.0} ms");
    }
    let mut ts = Vec::new();
    for _ in 0..n {
        tokio::time::sleep(gap).await;
        ts.push(one(size).await);
    }
    let w = &mut ts[2..].to_vec();
    w.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "size={size} gap={}ms prompt={prompt} iw={iw} n={} median={:.0} ms min={:.0} max={:.0} rtt={:?} cwnd={}",
        gap.as_millis(), w.len(), w[w.len() / 2], w[0], w[w.len() - 1],
        conn.rtt(), conn.stats().path.cwnd
    );
}

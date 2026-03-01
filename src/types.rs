pub type TierId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub tier: TierId,
    pub group: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(tier: TierId, group: u32, expert: u32) -> Self {
        Self {
            tier,
            group,
            expert,
        }
    }

    pub fn encode(self) -> String {
        format!("{}:{}:{}", self.tier, self.group, self.expert)
    }

    pub fn parse(value: &str) -> Option<Self> {
        let parts: Vec<&str> = value.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        Some(Self {
            tier: parts[0].parse().ok()?,
            group: parts[1].parse().ok()?,
            expert: parts[2].parse().ok()?,
        })
    }
}

pub fn fnv64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

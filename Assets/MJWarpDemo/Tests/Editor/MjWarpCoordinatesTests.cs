using NUnit.Framework;
using UnityEngine;

namespace MJWarpDemo.Tests
{
    public sealed class MjWarpCoordinatesTests
    {
        [Test]
        public void PositionSwizzlesYAndZ()
        {
            Assert.That(MjWarpCoordinates.Position(new[] { 1f, 2f, 3f }), Is.EqualTo(new Vector3(1f, 3f, 2f)));
        }

        [Test]
        public void IdentityQuaternionUsesOfficialMuJoCoUnityMapping()
        {
            Quaternion converted = MjWarpCoordinates.Rotation(new[] { 1f, 0f, 0f, 0f });
            Assert.That(converted.x, Is.EqualTo(0f));
            Assert.That(converted.y, Is.EqualTo(0f));
            Assert.That(converted.z, Is.EqualTo(0f));
            Assert.That(Mathf.Abs(converted.w), Is.EqualTo(1f));
        }

        [Test]
        public void BoxSizeUsesTheSameAxisSwizzle()
        {
            Assert.That(MjWarpCoordinates.Size(2f, 4f, 6f), Is.EqualTo(new Vector3(2f, 6f, 4f)));
        }
    }
}
